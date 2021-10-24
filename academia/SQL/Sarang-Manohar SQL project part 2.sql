use orders;

/* Q7. Write a query to display carton id, (len*width*height) as carton_vol and identify the optimum carton 
(carton with the least volume whose volume is greater than the total volume of all items (len * width * height * product_quantity)) 
for a given order whose order id is 10006, Assume all items of an order are packed into one single carton (box). 
(1 ROW) [NOTE: CARTON TABLE] */
select carton_id, len*width*height as carton_vol 
from CARTON 
where len*width*height>(
	select sum(PRODUCT_QUANTITY*LEN*WIDTH*HEIGHT) as total_prod_volume
	from ORDER_ITEMS as oi 
		left outer join PRODUCT as prd
		on oi.product_id = prd.product_id
	where order_id = '10006') 
order by len*width*height limit 1;

/* Q8. Write a query to display details (customer id,customer fullname,order id,product quantity) of customers 
who bought more than ten (i.e. total order qty) products per shipped order. (11 ROWS) 
[NOTE: TABLES TO BE USED - online_customer, order_header, order_items,]*/
select cust.customer_id, concat(customer_fname,' ',customer_lname) as customer_fullname, ordr_itm.order_id, total_order_quantity
from ONLINE_CUSTOMER cust
	left outer join ORDER_HEADER ordr_head
    on cust.customer_id = ordr_head.customer_id
    inner join (select order_id, sum(product_quantity) as total_order_quantity from ORDER_ITEMS group by order_id having sum(product_quantity)>10) ordr_itm
    on ordr_head.order_id = ordr_itm.order_id
where ordr_head.order_status = 'Shipped'
order by 1;

/*
9. Write a query to display the order_id, customer id and cutomer full name of customers 
along with (product_quantity) as total quantity of products shipped for order ids > 10060. 
(6 ROWS) [NOTE: TABLES TO BE USED - online_customer, order_header, order_items]*/
select ordr_itm.order_id, cust.customer_id, concat(customer_fname,' ',customer_lname) as customer_fullname, total_order_quantity
	from ONLINE_CUSTOMER cust
	left outer join ORDER_HEADER ordr_head
    on cust.customer_id = ordr_head.customer_id
    inner join (select order_id, sum(product_quantity) as total_order_quantity from ORDER_ITEMS where order_id > 10060 group by order_id) ordr_itm
    on ordr_head.order_id = ordr_itm.order_id
where ordr_head.order_status = 'Shipped'
order by ordr_itm.order_id;

/*
10. Write a query to display product class description ,total quantity (sum(product_quantity),Total value (product_quantity * product price) 
and show which class of products have been shipped highest(Quantity) to countries outside India other than USA? 
Also show the total value of those items. 
(1 ROWS)[NOTE:PRODUCT TABLE,ADDRESS TABLE,ONLINE_CUSTOMER TABLE,ORDER_HEADER TABLE,ORDER_ITEMS TABLE,PRODUCT_CLASS TABLE]*/
select country, PRODUCT_CLASS_DESC,prd.PRODUCT_CLASS_CODE, sum(product_quantity) total_quantity, sum(product_quantity*product_price) total_value
from ORDER_ITEMS ordr_itm 
	left outer join PRODUCT prd
    on ordr_itm.product_id = prd.product_id
    left outer join PRODUCT_CLASS prd_cls
    on prd.PRODUCT_CLASS_CODE = prd_cls.PRODUCT_CLASS_CODE
    inner join ORDER_HEADER ordr_head
    on ordr_itm.order_id = ordr_head.order_id
    and ordr_head.order_status = 'Shipped'
    left outer join ONLINE_CUSTOMER cust
    on ordr_head.customer_id = cust.customer_id
    inner join ADDRESS addr
    on cust.address_id = addr.address_id
    and addr.country not in ('India', 'USA')
group by country, PRODUCT_CLASS_DESC,prd.PRODUCT_CLASS_CODE
order by total_quantity desc limit 1;